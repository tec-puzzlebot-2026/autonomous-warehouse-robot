// Drive the original MotorTest RTL through its UART and encoder pins.
#include "VMotorTest.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef TANG_CLOCK_HZ
#define TANG_CLOCK_HZ 2304000
#endif

class Tx {
public:
    explicit Tx(int baud) : period((TANG_CLOCK_HZ + baud / 2) / baud) {}
    void append(const std::string& hex) {
        if (hex == "-") return;
        if (hex.size() % 2) throw std::runtime_error("odd serial hex length");
        for (std::size_t i = 0; i < hex.size(); i += 2) {
            auto byte = static_cast<unsigned>(std::stoul(hex.substr(i, 2), nullptr, 16));
            pending.push_back((1u << 9) | (byte << 1));
        }
    }
    bool tick() {
        if (remaining == 0) {
            if (bit == 10) {
                if (pending.empty()) return true;
                frame = pending.front();
                pending.pop_front();
                bit = 0;
            }
            value = (frame >> bit++) & 1u;
            remaining = period;
        }
        --remaining;
        return value;
    }

private:
    const int period;
    std::deque<unsigned> pending;
    unsigned frame = 0;
    int bit = 10;
    int remaining = 0;
    bool value = true;
};

class Rx {
public:
    explicit Rx(int baud) : period((TANG_CLOCK_HZ + baud / 2) / baud) {}
    void tick(bool high, std::ostringstream& out) {
        if (bit < 0) {
            if (!high) {
                bit = 0;
                value = 0;
                remaining = period + period / 2;
            }
        } else if (--remaining == 0) {
            if (bit < 8) {
                value |= static_cast<unsigned>(high) << bit++;
                remaining = period;
            } else {
                if (high) out << std::setw(2) << std::setfill('0') << std::hex << value;
                bit = -1;
            }
        }
    }

private:
    const int period;
    int bit = -1;
    int remaining = 0;
    unsigned value = 0;
};

int main() {
    VMotorTest dut;
    Tx usb(115200);
    Tx nano(9600);
    Rx reply(115200);
    const std::array<unsigned, 4> gray{0, 1, 3, 2};
    std::int64_t previous = 0;
    unsigned phase = 0;
    dut.clk = 0;
    dut.uart_rx_pin = dut.ir_rx_pin = 1;
    dut.enc_a = dut.enc_b = 0;
    dut.eval();
    std::cout << "READY " << TANG_CLOCK_HZ << std::endl;
    int cycles;
    std::int64_t position;
    std::string usb_hex, nano_hex;
    try {
        while (std::cin >> cycles >> position >> usb_hex >> nano_hex) {
            auto delta = position - previous;
            auto edges = std::llabs(delta);
            if (cycles <= 0 || cycles > TANG_CLOCK_HZ || edges * 4 >= cycles)
                throw std::runtime_error("invalid step or encoder rate");
            usb.append(usb_hex);
            nano.append(nano_hex);
            std::ostringstream data;
            std::int64_t accumulator = 0;
            int active = 0;
            for (int i = 0; i < cycles; ++i) {
                accumulator += edges;
                if (accumulator >= cycles) {
                    phase = (phase + (delta > 0 ? 1 : 3)) % 4;
                    accumulator -= cycles;
                }
                dut.enc_a = gray[phase] >> 1;
                dut.enc_b = gray[phase] & 1;
                dut.uart_rx_pin = usb.tick();
                dut.ir_rx_pin = nano.tick();
                dut.clk = 0;
                dut.eval();
                dut.clk = 1;
                dut.eval();
                reply.tick(dut.usb_uart_tx, data);
                if (dut.enb) active += static_cast<int>(dut.in4) - static_cast<int>(dut.in3);
            }
            previous = position;
            const auto result = data.str();
            std::cout << active << ' ' << (result.empty() ? "-" : result) << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    dut.final();
}
